use `NetEase`

DELIMITER //

DROP PROCEDURE IF EXISTS get_tagged_tracks//
CREATE PROCEDURE get_tagged_tracks()
BEGIN
	DECLARE total_num INT;
	DECLARE playlist_id INT;
	DECLARE track_id INT;
	DECLARE vuser_id INT;
	DECLARE tag VARCHAR(20);
	DECLARE track_ids VARCHAR(4000);
	DECLARE vtags VARCHAR(50);
	DECLARE tmpn1,tmpn2,i,j INT;

	-- 定义结束标志
	DECLARE done INT DEFAULT(FALSE);
	-- 定义游标
	DECLARE mycursor CURSOR FOR
		SELECT id, tags, user_id, tracks FROM playlists;
	-- 将结束标志与游标状态绑定
	DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;
	
	OPEN mycursor;
	-- total_num = 0;
	myloop: LOOP
		FETCH mycursor INTO playlist_id, vtags, vuser_id, track_ids;
		IF vtags='()' THEN ITERATE myloop;
		END IF;
		
		IF done THEN LEAVE myloop; 
		END IF; 
		
		SET i = 0;
		SET tmpn1 = count_str_tuples(vtags,',');
		SELECT playlist_id,vtags;
		
		WHILE i<tmpn1 DO
			SET i = i+1;
			SET tag = split(vtags,',',i);
			
			SET tmpn2 = count_str_tuples(track_ids,',');
			SET j = 0;
			WHILE j<tmpn2 DO
				SET j = j+1;
				-- SET total_num += 1;
				SET track_id = split(track_ids,',',j);
				INSERT IGNORE INTO tagged_tracks VALUES (track_id,tag,vuser_id,playlist_id);
			END WHILE;
		END WHILE;

	END LOOP myloop;
	CLOSE mycursor;
END//

DELIMITER ;